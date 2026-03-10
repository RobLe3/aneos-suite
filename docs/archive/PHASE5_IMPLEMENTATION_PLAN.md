# Phase 5 Gap Closure — Execution Plan

## Context

Phase 4 closed 14 gaps, leaving 8 open (1 P1, 5 P2, 1 P2-deferred, 1 P3-deferred).
This plan addresses all 6 non-deferred gaps in priority order. Two tasks are explicitly
deferred again: G-015 (ML classifier) and G-023 (menu decomposition).

**Input**: `docs/architecture/GAP_ANALYSIS.md` v3.0
**Test baseline**: 55 pass, 0 fail — must remain 0 fail after every sub-phase.
**Deferred**: 5E.1 (ML classifier), 5E.2 (menu decomposition)

---

## Sub-Phase Sequence

```
5A (Scientific Integrity) → 5B (API Consolidation) → 5C (Domain Model)
→ 5D (Security) → 5E (Deferred — skip)
```

Each sub-phase ends with `pytest tests/ -q` before proceeding.

---

## Phase 5A — Scientific Integrity

### Task 5A.1 — Fix Horizons NAIF Spacecraft Fetch (G-019 residual)

**File**: `aneos_core/datasets/ground_truth_dataset_preparation.py`

**Problem**: `_fetch_from_horizons()` splits each line on whitespace and looks for
tokens containing `=`. Horizons elements output uses `KEY = VALUE` with spaces around
`=`, so splitting on whitespace produces `["EC=", "0.551...", "QR=", ...]` — the key
token ends with `=` and the value token has no `=`, so the regex never fires. `A` is
never found; all 6 spacecraft return `None`.

**Fix**: Replace the parser with a regex that handles `KEY = VALUE` with optional
spaces. Read only lines inside the `$$SOE`/`$$EOE` ephemeris block.

Replace `_fetch_from_horizons` starting at line 219:

```python
def _fetch_from_horizons(self, naif_id: str, designation: str) -> Optional[Dict]:
    """Fetch osculating elements for a spacecraft from the JPL Horizons REST API."""
    import re
    try:
        resp = requests.get(self.HORIZONS_API, params={
            "format": "json", "COMMAND": f"'{naif_id}'", "OBJ_DATA": "NO",
            "MAKE_EPHEM": "YES", "EPHEM_TYPE": "ELEMENTS", "CENTER": "500@10",
            "START_TIME": "2020-01-01", "STOP_TIME": "2020-01-02",
            "STEP_SIZE": "1d", "OUT_UNITS": "AU-D",
        }, timeout=15)
        resp.raise_for_status()
        result_text = resp.json().get("result", "")
    except Exception as exc:
        self.logger.warning(f"Horizons fetch failed for {designation}: {exc}")
        return None

    # Parse all KEY = VALUE pairs from the $$SOE..$$EOE element block.
    # Horizons format: "EC= 5.55E-01 QR= 4.48E-01 IN= 9.18E+01 ..."
    # Keys and values are separated by " = " with possible whitespace variants.
    elements = {}
    in_block = False
    kv_pattern = re.compile(r'([A-Z]+)\s*=\s*([\d.Ee+\-]+)')
    for line in result_text.splitlines():
        if "$$SOE" in line:
            in_block = True
            continue
        if "$$EOE" in line:
            break
        if not in_block:
            continue
        for match in kv_pattern.finditer(line):
            try:
                elements[match.group(1)] = float(match.group(2))
            except ValueError:
                pass

    # Horizons element keys: A=semi-major axis, EC=eccentricity, IN=inclination,
    # OM=lon. asc. node, W=arg. periapsis, MA=mean anomaly
    if "A" not in elements:
        self.logger.warning(
            f"Horizons returned no elements for {designation} (NAIF {naif_id}). "
            "Response may be empty or in unexpected format."
        )
        return None

    return {
        "orbital_elements": {
            "a":     elements.get("A",  0.0),
            "e":     elements.get("EC", 0.0),
            "i":     elements.get("IN", 0.0),
            "omega": elements.get("OM", 0.0),
            "w":     elements.get("W",  0.0),
            "M":     elements.get("MA", 0.0),
        },
        "source": "JPL Horizons",
        "fetch_date": datetime.now().date().isoformat(),
    }
```

**Also add** a unit test that validates the parser against a known Horizons text sample.
**Create** `tests/test_horizons_parser.py`:

```python
"""Verify _fetch_from_horizons element parser handles Horizons text format."""
from aneos_core.datasets.ground_truth_dataset_preparation import GroundTruthDatasetBuilder
from unittest.mock import patch, MagicMock

# Minimal Horizons ELEMENTS response text (real format sample)
_SAMPLE_RESULT = """
Ephemeris / WWW_USER Sat Jan  1 00:00:00 2020 Pasadena, USA  ...
*******************************************************************************
$$SOE
2458849.500000000 = A.D. 2020-Jan-01 00:00:00.0000 TDB
 EC= 5.552003124456866E-01 QR= 4.481977069034823E-01 IN= 9.180426018781726E+01
 OM= 4.679244527993226E+01 W = 1.769695476793039E+02 Tp=  2458840.553734498937
 N = 7.165898831665516E-01 MA= 6.360781640547069E+00 TA= 2.013521505226268E+01
 A = 1.002248618219024E+00 AD= 1.556299529534565E+00 PR= 5.024055849499397E+02
$$EOE
"""

def test_parser_extracts_elements():
    builder = GroundTruthDatasetBuilder.__new__(GroundTruthDatasetBuilder)
    builder.logger = __import__('logging').getLogger(__name__)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"result": _SAMPLE_RESULT}
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp):
        result = builder._fetch_from_horizons("-31", "Voyager 1")

    assert result is not None, "Parser returned None — element extraction failed"
    oe = result["orbital_elements"]
    assert abs(oe["a"] - 1.002248618219024) < 1e-6, f"a mismatch: {oe['a']}"
    assert abs(oe["e"] - 0.5552003124456866) < 1e-6, f"e mismatch: {oe['e']}"
    assert abs(oe["i"] - 91.80426018781726) < 1e-6, f"i mismatch: {oe['i']}"
```

Add to `.gitignore` exceptions: `!tests/test_horizons_parser.py`

**Verify**:
```python
from aneos_core.datasets.ground_truth_dataset_preparation import GroundTruthDatasetBuilder
b = GroundTruthDatasetBuilder()
b.compile_verified_artificial_objects()
fetched = [o for o in b.artificial_objects if "JPL Horizons" in o.source and "fallback" not in o.source]
print(f"Objects with real Horizons data: {len(fetched)}")
# Expected: ≥ 1 (likely 4–6 depending on which NAIF IDs Horizons returns elements for)
# Interstellar probes (Pioneer 10/11, Voyager 1/2) should return hyperbolic elements (e > 1)
```

---

## Phase 5B — API Consolidation

### Task 5B.1 — Consolidate AnalysisResponse: Retire models.py Copy (G-012 residual)

**Problem**: `aneos_api/models.py:103` defines `AnalysisResponse(APIResponse)`.
`aneos_api/schemas/analysis.py` defines a separate `AnalysisResponse(BaseModel)`.
Endpoints use the `models.py` version; the `schemas/` version is dead code.

**Decision**: Keep `aneos_api/models.py` as the canonical location (endpoints
already import from it; it has the `APIResponse` base class pattern). Remove the
duplicate from `schemas/analysis.py` and re-export from `models.py`.

**Step 1** — Update `aneos_api/schemas/analysis.py`. Replace the `AnalysisResponse`
class body with a re-export that keeps the import path working:

```python
# aneos_api/schemas/analysis.py
"""Analysis response schema — canonical definition in aneos_api.models."""
from aneos_api.models import AnalysisResponse  # noqa: F401

__all__ = ["AnalysisResponse"]
```

Keep `IndicatorScores` in `schemas/analysis.py` — it is genuinely new and not in
`models.py`. Any endpoint or test that imports `IndicatorScores` from here continues
to work.

**Step 2** — Wire `DetectionResponse` from `schemas/detection.py` into the detection
endpoint if one exists. Search:
```bash
grep -rn "DetectionResponse\|/detect" aneos_api/endpoints/
```
If no detection endpoint exists, skip. If it does, add:
```python
from ..schemas.detection import DetectionResponse
```
and add `response_model=DetectionResponse` to the relevant route.

**Step 3** — Wire `HealthResponse` into the monitoring `/health` endpoint. Find
the health route in `aneos_api/endpoints/monitoring.py`:
```bash
grep -n "health\|preflight" aneos_api/endpoints/monitoring.py
```
Add:
```python
from ..schemas.health import HealthResponse
```
Add `response_model=HealthResponse` to the health route decorator.

**Step 4** — Run `make spec` to regenerate `docs/api/openapi.json`. Commit the result.

**Verify**:
```bash
grep -rn "from aneos_api.models import AnalysisResponse\|from ..models import.*AnalysisResponse" aneos_api/
# All references should point to models.py (one canonical location)
grep -rn "class AnalysisResponse" aneos_api/
# Expected: exactly 1 result (models.py:103)
pytest tests/ -q  # 0 fail
```

---

### Task 5B.2 — Verify and Fix NEODyS/MPC Endpoints (G-021 residual)

**File**: `aneos_core/data/sources/neodys.py`, `aneos_core/data/sources/mpc.py`

**Step 1 — Test the current implementations live**:
```python
from aneos_core.data.sources.neodys import NEODySSource
from aneos_core.data.sources.mpc import MPCSource
cfg = {}

n = NEODySSource(cfg)
result_n = n._make_request("99942")   # Apophis
print("NEODyS:", result_n)

m = MPCSource(cfg)
result_m = m._make_request("99942")
print("MPC:", result_m)
```

**Step 2 — Fix NEODyS `_make_request`** if the current URL fails. Replace with the
NEODyS-2 object API (returns JSON):

```python
def _make_request(self, designation: str) -> Optional[Dict]:
    import requests as _req
    # NEODyS-2 REST API — returns JSON orbital solution
    clean = designation.strip()
    url = f"https://newton.spacedys.com/neodys2/objects/{clean}/orbits/nominal"
    try:
        resp = _req.get(url, timeout=10,
                        headers={"User-Agent": "aNEOS/0.7 scientific research",
                                 "Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        # NEODyS-2 JSON: {"a": ..., "e": ..., "i": ..., "node": ..., "peri": ..., "M": ...}
        if not data or "a" not in data:
            return None
        return {
            "a":     float(data["a"]),
            "e":     float(data["e"]),
            "i":     float(data["i"]),
            "omega": float(data.get("node", data.get("omega", 0))),
            "w":     float(data.get("peri", data.get("w", 0))),
            "M":     float(data.get("M", 0)),
            "source": "NEODyS-2",
        }
    except Exception as exc:
        self.logger.debug(f"NEODyS-2 request failed for {designation}: {exc}")
        return None
```

**Step 3 — Fix MPC `_make_request`** if the current URL fails. Use the MPC
orbit search API (documented endpoint):

```python
def _make_request(self, designation: str) -> Optional[Dict]:
    import requests as _req
    try:
        resp = _req.get(
            "https://www.minorplanetcenter.net/search_orbits",
            params={"object_designation": designation, "format": "json"},
            timeout=10,
            headers={"User-Agent": "aNEOS/0.7 scientific research"},
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        obj = data[0] if isinstance(data, list) else data
        return {
            "a":     float(obj.get("a",    0)),
            "e":     float(obj.get("e",    0)),
            "i":     float(obj.get("i",    0)),
            "omega": float(obj.get("Node", obj.get("omega", 0))),
            "w":     float(obj.get("Peri", obj.get("w", 0))),
            "M":     float(obj.get("M",    0)),
            "H":     float(obj["H"]) if obj.get("H") else None,
            "source": "MPC",
        }
    except Exception as exc:
        self.logger.debug(f"MPC request failed for {designation}: {exc}")
        return None
```

**Step 4 — Add network integration tests** (skipped in normal CI runs):
**Create** `tests/test_data_sources_network.py`:

```python
"""Network integration tests for data source _make_request implementations.

Run with: pytest tests/test_data_sources_network.py -m network -v
Skip in CI: these tests require live API connectivity.
"""
import pytest

@pytest.mark.network
def test_neodys_returns_apophis():
    from aneos_core.data.sources.neodys import NEODySSource
    src = NEODySSource({})
    result = src._make_request("99942")  # Apophis
    assert result is not None, "NEODyS returned None for Apophis"
    assert 0 < result["a"] < 2.0, f"Apophis a={result['a']} out of expected range"
    assert 0 < result["e"] < 1.0, f"Apophis e={result['e']} out of expected range"

@pytest.mark.network
def test_mpc_returns_apophis():
    from aneos_core.data.sources.mpc import MPCSource
    src = MPCSource({})
    result = src._make_request("99942")
    assert result is not None, "MPC returned None for Apophis"
    assert 0 < result["a"] < 2.0, f"Apophis a={result['a']} out of expected range"
```

Add to `.gitignore` exceptions: `!tests/test_data_sources_network.py`

Add to `pytest.ini` (or `pyproject.toml`):
```ini
[pytest]
markers =
    network: marks tests as requiring live network access (deselect with -m 'not network')
```

**Verify**:
```bash
# Offline suite — must still pass
pytest tests/ -q -m "not network"
# Network suite — run manually when connected
pytest tests/test_data_sources_network.py -m network -v
```

---

### Task 5B.3 — Fix CI Spec Drift Check (G-027)

**File**: `Makefile`, `.github/workflows/ci.yml`, `CONTRIBUTING.md`

**Step 1 — Make spec output deterministic**. In `Makefile`, replace the `spec` target
with a version that sorts JSON keys:

```makefile
spec:
	@mkdir -p docs/api
	python -c "\
import sys, json; sys.path.insert(0, '.'); \
from aneos_api.app import create_app; \
app = create_app(); \
print(json.dumps(app.openapi(), indent=2, sort_keys=True))" \
	> docs/api/openapi.json
	@echo "OpenAPI spec written to docs/api/openapi.json"
```

**Step 2 — Improve CI failure message**. In `.github/workflows/ci.yml`, replace the
spec verification step with:

```yaml
- name: Verify OpenAPI spec is up to date
  run: |
    make spec
    if ! git diff --exit-code docs/api/openapi.json; then
      echo ""
      echo "ERROR: docs/api/openapi.json is out of date."
      echo "Run 'make spec' locally and commit the updated file."
      exit 1
    fi
```

**Step 3 — Add CONTRIBUTING.md section** (create if absent):

```markdown
## API Development Workflow

When you add or modify a FastAPI endpoint, regenerate the OpenAPI specification
before committing:

```bash
make spec
git add docs/api/openapi.json
git commit -m "chore: regenerate openapi.json"
```

CI will fail with a clear error message if the spec is stale.
```

**Verify**:
```bash
make spec                             # runs cleanly
git diff docs/api/openapi.json        # no changes (already up to date after sort_keys)
pytest tests/ -q                      # 0 fail
```

---

## Phase 5C — Domain Model Completion

### Task 5C.1 — Thread PhysicalProperties Through Impact Calculator (G-011 residual)

**CAUTION: broadest refactor in Phase 5. Run `pytest tests/ -q` before starting and after
each step.**

**Approach**: Add a private helper `_get_diameter_km()` to `ImpactProbabilityCalculator`
that checks `PhysicalProperties` first, then falls back to `OrbitalElements.diameter`.
Thread an optional `physical_props` parameter through the 8 private methods that read
`orbital_elements.diameter`. Do NOT remove `OrbitalElements.diameter` yet — that is a
Phase 6 change after all callers are verified.

**Step 1 — Read the full ImpactProbabilityCalculator class header** to confirm it has
access to `NEOData` at call time:
```bash
grep -n "def calculate_impact\|def _calculate\|NEOData\|physical_props" \
    aneos_core/analysis/impact_probability.py | head -30
```

**Step 2 — Add helper method** to `ImpactProbabilityCalculator`. Find the class's
first method and insert before it:

```python
@staticmethod
def _get_diameter_km(
    orbital_elements: "OrbitalElements",
    physical_props: Optional["PhysicalProperties"] = None,
) -> Optional[float]:
    """Return diameter in km from physical_props if available, else orbital_elements."""
    if physical_props is not None and getattr(physical_props, "diameter_km", None):
        return physical_props.diameter_km
    return getattr(orbital_elements, "diameter", None) or None
```

**Step 3 — Update the 8 private methods** that read `orbital_elements.diameter`.
For each method signature, add `physical_props: Optional["PhysicalProperties"] = None`.
Replace every `orbital_elements.diameter` read with
`self._get_diameter_km(orbital_elements, physical_props)`.

Target lines (confirmed by audit):
- `_calculate_impact_energy(self, orbital_elements)` → lines 601, 605
- `_calculate_crater_diameter_proper(self, orbital_elements)` → lines 688, 690
- The method at line 965 → line 965
- `_calculate_crater_diameter_proper` second occurrence → lines 1304, 1308, 1353, 1355

Example transformation for `_calculate_impact_energy`:
```python
# BEFORE
def _calculate_impact_energy(self, orbital_elements: OrbitalElements) -> Optional[float]:
    if not orbital_elements.diameter:
        return None
    diameter_m = orbital_elements.diameter * 1000

# AFTER
def _calculate_impact_energy(
    self,
    orbital_elements: OrbitalElements,
    physical_props: Optional["PhysicalProperties"] = None,
) -> Optional[float]:
    diameter_km = self._get_diameter_km(orbital_elements, physical_props)
    if not diameter_km:
        return None
    diameter_m = diameter_km * 1000
```

**Step 4 — Update callers** inside `ImpactProbabilityCalculator` to pass
`physical_props` when the containing method has access to `NEOData`:
```bash
grep -n "_calculate_impact_energy\|_calculate_crater" aneos_core/analysis/impact_probability.py
```
For each internal call site, add `physical_props=neo_data.physical_properties` if
`neo_data` is in scope, else leave as-is (the helper falls back to `orbital_elements`).

**Step 5 — Fix thermal_ir_analysis.py** (3 callsites):
```
thermal_ir_analysis.py:487   diameter = neo_data.orbital_elements.diameter
thermal_ir_analysis.py:648   neo_data.orbital_elements.rot_per
thermal_ir_analysis.py:1354  diameter = neo_data.orbital_elements.diameter
```
For the two `diameter` reads:
```python
# BEFORE
diameter = neo_data.orbital_elements.diameter
# AFTER
pp = getattr(neo_data, "physical_properties", None)
diameter = (pp.diameter_km if pp and pp.diameter_km else neo_data.orbital_elements.diameter)
```

For the `rot_per` read (line 648):
```python
# BEFORE
neo_data.orbital_elements.rot_per if hasattr(neo_data, 'orbital_elements') and neo_data.orbital_elements else 12.0
# AFTER
pp = getattr(neo_data, "physical_properties", None)
(pp.rotation_period if pp and pp.rotation_period else
 (neo_data.orbital_elements.rot_per if neo_data.orbital_elements else 12.0))
```

**Step 6 — Run full test suite**:
```bash
pytest tests/ -q   # must remain 0 fail
```

**Verify no new callsites were missed**:
```bash
grep -rn "orbital_elements\.diameter\|orbital_elements\.albedo\|orbital_elements\.rot_per\|orbital_elements\.spectral_type" \
    aneos_core/ aneos_menu.py
# Expected: only the 3 fallback lines in indicators/physical.py remain
# (those already have the new physical_properties-first pattern)
```

---

## Phase 5D — Security Completion

### Task 5D.1 — Replace Pickle in ML Persistence Layer (G-026)

**Files**: `aneos_core/ml/models.py`, `aneos_core/ml/prediction.py`

**Step 1 — Add joblib import guard** at the top of `ml/models.py` (after existing
imports):

```python
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
```

**Step 2 — Replace `AnomalyDetectionModel.save_model()`** (lines 141–156):

```python
def save_model(self, filepath: str) -> None:
    """Save model to file using joblib (sklearn) + JSON (metadata)."""
    if not HAS_JOBLIB:
        raise ImportError("joblib is required for model persistence. pip install joblib")

    model_data = {
        'model': self.model,
        'scaler': self.scaler,
        'config': self.config,
        'feature_names': self.feature_names,
        'model_id': self.model_id,
        'is_trained': self.is_trained,
        'outlier_bounds': getattr(self, 'outlier_bounds', None),
    }
    joblib.dump(model_data, filepath)
    logger.info(f"Model saved to {filepath}")
```

**Step 3 — Replace `AnomalyDetectionModel.load_model()`** (lines 158–173):

```python
def load_model(self, filepath: str) -> None:
    """Load model from file (joblib format)."""
    if not HAS_JOBLIB:
        raise ImportError("joblib is required for model persistence. pip install joblib")

    model_data = joblib.load(filepath)
    self.model = model_data['model']
    self.scaler = model_data['scaler']
    self.config = model_data['config']
    self.feature_names = model_data['feature_names']
    self.model_id = model_data['model_id']
    self.is_trained = model_data['is_trained']
    if 'outlier_bounds' in model_data:
        self.outlier_bounds = model_data['outlier_bounds']
    logger.info(f"Model loaded from {filepath}")
```

**Step 4 — Replace `ModelEnsemble.save_ensemble()` metadata** (lines 597–609).
Replace `pickle.dump(metadata, f)` with JSON:

```python
# Replace:
metadata_path = ensemble_dir / "ensemble_metadata.pkl"
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

# With:
import json as _json
metadata_path = ensemble_dir / "ensemble_metadata.json"
with open(metadata_path, 'w') as f:
    _json.dump(metadata, f)
```

**Step 5 — Replace `ModelEnsemble.load_ensemble()` metadata** (lines 615–621):

```python
# Replace:
metadata_path = ensemble_dir / "ensemble_metadata.pkl"
with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

# With:
import json as _json
metadata_path = ensemble_dir / "ensemble_metadata.json"
# Support legacy .pkl filename for backward compat during migration
if not metadata_path.exists():
    metadata_path = ensemble_dir / "ensemble_metadata.pkl"
    if metadata_path.exists():
        import pickle as _pickle
        with open(metadata_path, 'rb') as f:
            metadata = _pickle.load(f)
        # Immediately re-save in JSON format
        json_path = ensemble_dir / "ensemble_metadata.json"
        with open(json_path, 'w') as f:
            _json.dump(metadata, f)
        metadata_path.unlink()   # remove legacy file
    else:
        raise FileNotFoundError(f"Ensemble metadata not found: {metadata_path}")
else:
    with open(metadata_path, 'r') as f:
        metadata = _json.load(f)
```

**Step 6 — Fix `ml/prediction.py`** (lines 143–150):

```python
# Replace:
import pickle
with open(metadata_file, 'rb') as f:
    metadata = pickle.load(f)

# With:
import json as _json
from pathlib import Path as _Path
# Try JSON first (new format), fall back to legacy pkl
json_file = _Path(str(metadata_file).replace('.pkl', '.json'))
if json_file.exists():
    with open(json_file, 'r') as f:
        metadata = _json.load(f)
elif metadata_file.exists():
    import pickle as _pickle
    with open(metadata_file, 'rb') as f:
        metadata = _pickle.load(f)
else:
    logger.warning(f"Ensemble metadata not found: {metadata_file}")
    return None
```

**Step 7 — Remove `import pickle`** from `ml/models.py` (top of file). Verify:
```bash
grep -n "^import pickle" aneos_core/ml/models.py   # Expected: 0 results
grep -n "pickle" aneos_core/ml/models.py            # Expected: 0 results (only _pickle in migration block)
grep -n "^import pickle" aneos_core/ml/prediction.py  # Expected: 0 results
```

**Step 8**:
```bash
pytest tests/ -q   # 0 fail
```

---

## Phase 5E — Deferred

### Task 5E.1 — ML Classifier Activation (G-015) — DEFERRED

Requires `FeatureVector.from_ground_truth_object()`, scikit-learn training pipeline,
and `MLDetectorWrapper` registration in `DetectionManager`. Multi-day external work.
No action in Phase 5.

### Task 5E.2 — aneos_menu.py Decomposition (G-023) — DEFERRED

11,500-line monolith requires dedicated standalone phase with incremental extraction.
No action in Phase 5.

---

## Verification Checkpoints

**After 5A:**
```python
from aneos_core.datasets.ground_truth_dataset_preparation import GroundTruthDatasetBuilder
b = GroundTruthDatasetBuilder()
b.compile_verified_artificial_objects()
real = [o for o in b.artificial_objects if "JPL Horizons" in o.source and "fallback" not in o.source]
print(f"Objects with real Horizons data: {len(real)}")  # ≥ 1
pytest tests/ -q  # new test_horizons_parser.py passes
```

**After 5B:**
```bash
grep -rn "class AnalysisResponse" aneos_api/       # exactly 1 result (models.py)
pytest tests/ -m "not network" -q                   # 0 fail
make spec && git diff --exit-code docs/api/openapi.json  # clean
```

**After 5C:**
```bash
grep -rn "orbital_elements\.diameter" aneos_core/analysis/impact_probability.py
# Expected: 0 results
grep -rn "orbital_elements\.rot_per\|orbital_elements\.diameter" aneos_core/validation/thermal_ir_analysis.py
# Expected: 0 results
pytest tests/ -q  # 0 fail
```

**After 5D:**
```bash
grep -n "^import pickle" aneos_core/ml/models.py aneos_core/ml/prediction.py
# Expected: 0 results
pytest tests/ -q  # 0 fail
```

---

## Files Modified / Created

| Action | Path | Task |
|--------|------|------|
| MODIFY | `aneos_core/datasets/ground_truth_dataset_preparation.py` | 5A.1 |
| CREATE | `tests/test_horizons_parser.py` | 5A.1 |
| MODIFY | `aneos_api/schemas/analysis.py` | 5B.1 |
| MODIFY | `aneos_api/endpoints/monitoring.py` | 5B.1 |
| MODIFY | `docs/api/openapi.json` | 5B.1, 5B.3 |
| MODIFY | `aneos_core/data/sources/neodys.py` | 5B.2 |
| MODIFY | `aneos_core/data/sources/mpc.py` | 5B.2 |
| CREATE | `tests/test_data_sources_network.py` | 5B.2 |
| MODIFY | `pytest.ini` or `pyproject.toml` | 5B.2 |
| MODIFY | `Makefile` | 5B.3 |
| MODIFY | `.github/workflows/ci.yml` | 5B.3 |
| CREATE | `CONTRIBUTING.md` | 5B.3 |
| MODIFY | `aneos_core/analysis/impact_probability.py` | 5C.1 |
| MODIFY | `aneos_core/validation/thermal_ir_analysis.py` | 5C.1 |
| MODIFY | `aneos_core/ml/models.py` | 5D.1 |
| MODIFY | `aneos_core/ml/prediction.py` | 5D.1 |
| MODIFY | `.gitignore` | 5A.1, 5B.2 |
