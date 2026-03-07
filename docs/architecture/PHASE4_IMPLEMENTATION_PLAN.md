# Phase 4 Implementation Plan — Gap Closure

**Version**: 1.0
**Date**: 2026-03-07
**Input**: GAP_ANALYSIS.md v2.0
**Prior phases**: 1 (infrastructure), 2 (physical indicators + ground truth),
3 (real data wiring + honest UI + G-001 fix)

Gaps to close: G-004(residual), G-008, G-009, G-011, G-012, G-013, G-014,
G-015, G-016, G-017, G-018, G-019, G-020, G-021, G-022, G-023, G-024, G-025

Dependencies drive the sequence. Each sub-phase has a verification checkpoint
before the next sub-phase starts.

---

## Rationale for ordering

- **4A (Scientific Accuracy)**: G-019/G-020 are P1 scientific validity gaps.
  Expanding the corpus and documenting the Bayesian ceiling are prerequisites
  for any published accuracy claim. G-004 startup acknowledgement is a one-file
  UX fix that unblocks confident offline use.
- **4B (Data Sources)**: NEODyS/MPC/Horizons non-functional (G-021) silently
  degrades enrichment quality for every analysis. G-024 temporal analysis
  depends on Horizons being wired.
- **4C (Architecture)**: Detector archive (G-008), test harness move (G-013),
  and EMERGENCY suppression fix (G-009/G-022) are all independent cleanup
  tasks. Do them together in one pass.
- **4D (API Quality)**: DTO schemas (G-012) and OpenAPI spec (G-014) are
  coupled; do together. OrbitalElements separation (G-011) has the broadest
  refactor surface — do last in this sub-phase with careful test-before/after.
- **4E (Future Capabilities)**: ML (G-015), Redis (G-016), pickle (G-017),
  chunk boundary (G-018), AI annotator (G-025), and menu decomposition (G-023)
  are independent and can be parallelised or deferred.

---

## Phase 4A — Scientific Accuracy

### Task 4A.1 — Expand Artificial Object Corpus (G-019)

**File:** `aneos_core/datasets/ground_truth_dataset_preparation.py`

The current corpus has 3 confirmed artificials. The Implementation Plan target
was ≥ 8. Spacecraft not in SBDB can be fetched from JPL Horizons via NAIF IDs
using the Horizons REST API directly (simpler than going through HorizonsSource).

**Step 1 — Add NAIF-keyed object definitions**

Add a new constant below `ARTIFICIAL_PHYSICAL_FALLBACK`:

```python
# Confirmed heliocentric artificial objects fetchable from JPL Horizons by NAIF ID.
# Source: JPL NAIF/SPICE kernel registry + published mission documentation.
HORIZONS_ARTIFICIALS = {
    "Pioneer 10": {
        "naif_id": "-23",
        "fallback": {"a": 47.0, "e": 0.051, "i": 3.11, "omega": 75.7, "w": 130.7, "M": 0.0},
        "physical": {"diameter": 2.74, "mass_estimate": 259.0},  # m, kg
        "notes": "Pioneer 10 — heliocentric escape trajectory; last contact 2003"
    },
    "Pioneer 11": {
        "naif_id": "-24",
        "fallback": {"a": 31.0, "e": 0.056, "i": 17.1, "omega": 95.0, "w": 200.0, "M": 0.0},
        "physical": {"diameter": 2.74, "mass_estimate": 259.0},
        "notes": "Pioneer 11 — heliocentric escape; last contact 1995"
    },
    "Voyager 1": {
        "naif_id": "-31",
        "fallback": {"a": 150.0, "e": 0.059, "i": 35.7, "omega": 250.0, "w": 50.0, "M": 0.0},
        "physical": {"diameter": 3.7, "mass_estimate": 825.5},
        "notes": "Voyager 1 — interstellar space (a>100 AU); confirmed artificial"
    },
    "Voyager 2": {
        "naif_id": "-32",
        "fallback": {"a": 120.0, "e": 0.057, "i": 79.0, "omega": 46.0, "w": 210.0, "M": 0.0},
        "physical": {"diameter": 3.7, "mass_estimate": 825.5},
        "notes": "Voyager 2 — interstellar space; confirmed artificial"
    },
    "New Horizons": {
        "naif_id": "-98",
        "fallback": {"a": 46.0, "e": 0.057, "i": 2.25, "omega": 175.0, "w": 70.0, "M": 0.0},
        "physical": {"diameter": 2.2, "mass_estimate": 478.0},
        "notes": "New Horizons — post-Pluto flyby; confirmed artificial"
    },
    "DSCOVR": {
        "naif_id": "-227",
        "fallback": {"a": 1.001, "e": 0.004, "i": 0.15, "omega": 0.0, "w": 0.0, "M": 0.0},
        "physical": {"diameter": 2.0, "mass_estimate": 570.0},
        "notes": "DSCOVR at Earth-Sun L1; confirmed artificial"
    },
}
```

**Step 2 — Add `_fetch_from_horizons()` method**

Add below `_fetch_from_sbdb()` in `GroundTruthDatasetBuilder`:

```python
HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"

def _fetch_from_horizons(self, naif_id: str, designation: str) -> Optional[Dict]:
    """
    Fetch osculating heliocentric orbital elements from JPL Horizons for a
    spacecraft identified by NAIF integer ID.

    Uses OBJ_DATA=NO, MAKE_EPHEM=YES, EPHEM_TYPE=ELEMENTS, CENTER='500@10'
    (heliocentric, ecliptic J2000).
    """
    try:
        resp = requests.get(
            self.HORIZONS_API,
            params={
                "format": "json",
                "COMMAND": f"'{naif_id}'",
                "OBJ_DATA": "NO",
                "MAKE_EPHEM": "YES",
                "EPHEM_TYPE": "ELEMENTS",
                "CENTER": "500@10",   # heliocentric
                "START_TIME": "2020-01-01",
                "STOP_TIME": "2020-01-02",
                "STEP_SIZE": "1d",
                "OUT_UNITS": "AU-D",
            },
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        self.logger.warning(f"Horizons fetch failed for {designation} (NAIF {naif_id}): {exc}")
        return None

    result_text = data.get("result", "")
    # Parse the element block from the text output
    elements = {}
    for line in result_text.splitlines():
        # Horizons element lines look like: "  EC= 5.700000000000000E-02  QR= 0.9789..."
        parts = line.strip().split()
        for part in parts:
            if "=" in part:
                k, _, v = part.partition("=")
                try:
                    elements[k.strip()] = float(v.strip())
                except ValueError:
                    pass

    # Map Horizons keys to our internal format
    # EC=eccentricity, A=semi-major axis (AU), IN=inclination (deg),
    # OM=long ascending node, W=arg periapsis, MA=mean anomaly
    if "A" not in elements and "EC" not in elements:
        self.logger.warning(f"Horizons returned no parseable elements for {designation}")
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

**Step 3 — Add `compile_horizons_artificial_objects()` method**

```python
def compile_horizons_artificial_objects(self) -> List[GroundTruthObject]:
    """
    Compile spacecraft objects from JPL Horizons (not in SBDB minor-body catalogue).
    Falls back to hardcoded elements per object if Horizons is unreachable.
    """
    objects = []
    for name, spec in self.HORIZONS_ARTIFICIALS.items():
        fetched = self._fetch_from_horizons(spec["naif_id"], name)
        if fetched:
            orbital_elements = fetched["orbital_elements"]
            source = f"JPL Horizons NAIF {spec['naif_id']} (fetched {fetched['fetch_date']})"
        else:
            orbital_elements = dict(spec["fallback"])
            source = f"Hardcoded fallback (Horizons unavailable) — NAIF {spec['naif_id']}"

        obj = GroundTruthObject(
            object_id=name,
            is_artificial=True,
            orbital_elements=orbital_elements,
            physical_params=dict(spec["physical"]),
            source=source,
            verification_notes=spec["notes"],
        )
        objects.append(obj)

    self.artificial_objects.extend(objects)
    self.logger.info(f"Compiled {len(objects)} Horizons spacecraft objects")
    return objects
```

**Step 4 — Call from `compile_verified_artificial_objects()`**

At the end of `compile_verified_artificial_objects()`, before the final
`self.logger.info(...)` line, add:

```python
# Also fetch spacecraft not in SBDB (Pioneers, Voyagers, New Horizons, DSCOVR)
self.compile_horizons_artificial_objects()
artificial_objects = self.artificial_objects   # already extended above
```

**Verification:**
```python
from aneos_core.datasets.ground_truth_dataset_preparation import GroundTruthDatasetBuilder
b = GroundTruthDatasetBuilder()
b.compile_verified_artificial_objects()
print(len(b.artificial_objects))
# Expected: ≥ 9 (3 SBDB + 6 Horizons)
```

Note: Pioneer 10/11 and Voyager 1/2 are on escape trajectories with semi-major
axes > 30 AU. Their orbital anomaly scores will differ from inner-solar-system
spacecraft. This is expected — the ground truth set should be heterogeneous.

---

### Task 4A.2 — Document and Surface Bayesian Posterior Ceiling (G-020)

Three changes: README, detection output in aneos_menu, and sigma5 success criteria.

**Step 1 — Add interpretation section to README.md**

In the "Technical Specifications / Detection Capabilities" block, after the
"Statistical Framework" bullet, add:

```markdown
- **Detection Interpretation**: sigma-level (σ) measures statistical rarity
  under the natural NEO null hypothesis. P(artificial) is the Bayesian posterior
  incorporating a 0.1% base rate. A σ≥5 detection means the observation has
  a 1-in-1.74M chance of arising naturally; with a 0.1% prior this translates
  to P(artificial)≈1–5%. Propulsion signatures or course corrections are needed
  to push P(artificial) above 10%.
```

**Step 2 — Add interpretation line to detection output in aneos_menu.py**

Find the detection result display blocks (search for `sigma_confidence` or
`is_artificial` print statements in `single_neo_analysis`, `batch_analysis`,
`interactive_analysis`). After the sigma and Bayesian probability lines, add:

```python
self.console.print(
    "[dim]Interpretation: σ measures statistical rarity (null hypothesis probability).\n"
    "P(artificial) incorporates 0.1% base rate. Smoking-gun evidence "
    "(propulsion/manoeuvres) needed to push P(artificial) above 10%.[/dim]"
)
```

**Step 3 — Update sigma5_success_criteria.md**

Add a "Detection Interpretation Framework" section after the "Verification Log":

```markdown
## Detection Interpretation Framework

| Metric | Meaning | Bound |
|--------|---------|-------|
| Sigma level (σ) | Rarity of observation under natural NEO null hypothesis | Unbounded above 0 |
| P(artificial) | Bayesian posterior with 0.1% base prior | Max ~4% from orbital+physical evidence |
| Classification threshold | σ ≥ 5.0 → flagged as artificial | Corresponds to p < 5.7×10⁻⁷ |

Smoking-gun evidence (observed course corrections, propulsion burn signatures,
radar specular return inconsistent with natural body) is required to push
P(artificial) above 10%. The sigma-5 flag means "extremely unusual for a
natural object", not "probably artificial".
```

**Verification:** Re-run `test_tesla_roadster_detection` and confirm the
`bayesian_probability > 0.01` assertion still passes with the new documentation
in place (no code logic changed, only display/docs).

---

### Task 4A.3 — Make aneos_menu Startup Failure Acknowledged (G-004 residual)

**File:** `aneos_menu.py`, startup section (~line 115 where `_preflight` is set)

Currently:
```python
self._preflight = preflight_check()
# results stored but not acted upon
```

Replace with:

```python
self._preflight = preflight_check()
failed = [k for k, v in self._preflight.items() if v.get("status") == "error"]
critical = [k for k in failed if k in ("pipeline", "analysis")]
if critical:
    self.console.print("\n[bold red]SYSTEM PREFLIGHT FAILURES:[/bold red]")
    for k in critical:
        detail = self._preflight[k].get("detail", "unknown error")
        self.console.print(f"  [red]{k}[/red]: {detail}")
    self.console.print(
        "\n[yellow]Core components unavailable. Analysis functions will fail.[/yellow]"
        "\nContinue to menu anyway? [y/N] ",
        end=""
    )
    answer = input().strip().lower()
    if answer != "y":
        sys.exit(1)
```

This gates on user acknowledgement — does not silently continue, does not
break offline research (user can still type 'y').

**Verification:**
```bash
# With core unavailable (e.g., uninstalled aneos_core package temporarily):
python aneos_menu.py
# Expected: prints failure table and asks y/N before showing the menu
```

---

## Phase 4B — Data Source Completeness

### Task 4B.1 — Implement NEODyS `_make_request` (G-021)

**File:** `aneos_core/data/sources/neodys.py`

The current `_make_request` method has bare `pass` bodies (lines ~112–124).
NEODyS provides orbital elements via a simple query URL:
`https://newton.spacedys.com/neodys/index.php?pc=1.1.0&n={designation}&oc=10&m=10&nl=0&header=0`

Replace the `_make_request` stub:

```python
def _make_request(self, designation: str) -> Optional[Dict]:
    """Synchronous fetch from NEODyS for orbital elements."""
    import requests as _requests
    # NEODyS uses URL-encoded designation (spaces → %20, or use '+')
    clean_desig = designation.strip().replace(" ", "%20")
    url = f"https://newton.spacedys.com/neodys/index.php?pc=1.1.0&n={clean_desig}&oc=10&m=10&nl=0&header=0"
    try:
        resp = _requests.get(url, timeout=10,
                             headers={"User-Agent": "aNEOS/0.7 scientific research"})
        resp.raise_for_status()
        return {"raw": resp.text, "source": "NEODyS"}
    except Exception as exc:
        self.logger.debug(f"NEODyS request failed for {designation}: {exc}")
        return None
```

Also update `_parse_response()` / `fetch()` to extract `a`, `e`, `i` from the
NEODyS tab-separated element output. NEODyS returns lines like:
```
! Keplerian elements at epoch ...
a e i om w M
1.126 0.204 6.035 ...
```

Parse with:
```python
lines = [l for l in raw.splitlines() if not l.startswith("!") and l.strip()]
if lines:
    parts = lines[-1].split()
    # a, e, i, om, w, M — order varies; parse the header line to be safe
```

**Note**: NEODyS may not have all objects. Failures must be logged as DEBUG
and the source must return an empty result gracefully — not raise.

---

### Task 4B.2 — Implement MPC `_make_request` (G-021)

**File:** `aneos_core/data/sources/mpc.py`

MPC provides orbital elements via the MPC Web Services JSON API:
`https://minorplanetcenter.net/web_service/search_orbits?object_designation={designation}`

Replace the `_make_request` stub with:

```python
def _make_request(self, designation: str) -> Optional[Dict]:
    """Synchronous fetch from MPC orbit database."""
    import requests as _requests
    url = "https://minorplanetcenter.net/web_service/search_orbits"
    try:
        resp = _requests.get(url, params={"object_designation": designation},
                             timeout=10,
                             headers={"User-Agent": "aNEOS/0.7 scientific research",
                                      "Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        obj = data[0] if isinstance(data, list) else data
        return {
            "a": float(obj.get("a", 0)),
            "e": float(obj.get("e", 0)),
            "i": float(obj.get("i", 0)),
            "omega": float(obj.get("Node", 0)),
            "w": float(obj.get("Peri", 0)),
            "M": float(obj.get("M", 0)),
            "H": float(obj.get("H", 0)) if obj.get("H") else None,
            "source": "MPC",
        }
    except Exception as exc:
        self.logger.debug(f"MPC request failed for {designation}: {exc}")
        return None
```

---

### Task 4B.3 — Instantiate Horizons in DataFetcher (G-021)

**File:** `aneos_core/data/fetcher.py`

`HorizonsSource` exists but is never added to the source list in `__init__`.
Find where sources are instantiated (likely a list like `[SBDBSource(), ...]`).
Add `HorizonsSource()` to that list after SBDB.

```python
from .sources.horizons import HorizonsSource

# In DataFetcher.__init__():
self.sources = [
    SBDBSource(self.config),
    HorizonsSource(self.config),   # ADD THIS
    # NEODyS, MPC follow once _make_request is implemented
]
```

**Important**: Horizons requires `asyncio.run()` wrapping since it is async.
The existing `_fetch_from_source` wrapper already handles this via
`asyncio.run()` + `inspect.isawaitable()`. Verify this covers HorizonsSource.

**Verification after 4B.1–4B.3:**
```bash
grep -r "sources_used" neo_data/results/*.json | head -5
# Expected: results should show multiple sources, not just "SBDB"
```

---

### Task 4B.4 — Wire Orbital History to Horizons Ephemeris (G-024)

**File:** `aneos_menu.py`, `_generate_orbital_history()` (~line where it
currently returns `[current]`)

Replace the single-epoch return with a Horizons ephemeris call:

```python
def _generate_orbital_history(self, designation, base_elements):
    """Fetch multi-epoch orbital elements from JPL Horizons."""
    try:
        import requests
        resp = requests.get(
            "https://ssd.jpl.nasa.gov/api/horizons.api",
            params={
                "format": "json",
                "COMMAND": f"'{designation}'",
                "OBJ_DATA": "NO",
                "MAKE_EPHEM": "YES",
                "EPHEM_TYPE": "ELEMENTS",
                "CENTER": "500@10",
                "START_TIME": "2010-01-01",
                "STOP_TIME": "2030-01-01",
                "STEP_SIZE": "365d",
            },
            timeout=10
        )
        if resp.status_code == 200:
            history = _parse_horizons_elements(resp.json().get("result", ""))
            if history:
                return history
    except Exception as exc:
        self.logger.debug(f"Horizons orbital history failed for {designation}: {exc}")

    # Fallback: single current epoch
    current = {'epoch': 0, '_label': '[OBSERVED - current epoch only; Horizons unavailable]'}
    for k in ('a', 'e', 'i', 'om', 'w', 'M'):
        if k in base_elements:
            current[k] = base_elements[k]
    return [current]
```

Add helper `_parse_horizons_elements(result_text)` that extracts one row per
epoch from the Horizons text output. Return a list of dicts with `epoch`,
`a`, `e`, `i`, `om`, `w`, `M`, `_label: '[OBSERVED]'`.

**Verification:**
```python
# In a Python session with network access:
menu = AneosMenu()
hist = menu._generate_orbital_history("99942", {"a": 0.922, "e": 0.191, "i": 3.33})
print(len(hist), hist[0])
# Expected: multiple epochs with [OBSERVED] labels
```

---

## Phase 4C — Architecture Cleanup

### Task 4C.1 — Archive Superseded Detector Files (G-008)

**Step 1**: Create `aneos_core/detection/_archive/__init__.py` (empty).

**Step 2**: Move these files into `_archive/`:
- `sigma5_artificial_neo_detector.py`
- `corrected_sigma5_artificial_neo_detector.py`
- `production_artificial_neo_detector.py`
- `multimodal_sigma5_artificial_neo_detector.py`
- `sigma5_corrected_statistical_framework.py`

**Step 3**: In `aneos_core/detection/detection_manager.py`, remove or
comment out imports of the archived detectors. Keep only:
- `ValidatedSigma5ArtificialNEODetector` as priority 0
- Entries for any still-functional detectors (BASIC, etc.) can be kept with a
  `# DEPRECATED — use VALIDATED` comment if tests depend on them

**Step 4**: Run the full test suite and fix any import errors. Common breakages:
- `test_detection_manager.py` may import non-VALIDATED types
- `aneos_menu.py` may reference `DetectorType.MULTIMODAL`

Search and fix:
```bash
grep -rn "MULTIMODAL\|PRODUCTION\|CORRECTED\|BASIC" aneos_menu.py aneos_core/
```

**Verification:**
```bash
ls aneos_core/detection/
# Expected: __init__.py, _archive/, detection_manager.py, validated_sigma5_artificial_neo_detector.py
# (+ artificial_neo_test_suite.py until 4C.2)
pytest tests/ -q
# 54 pass, 0 fail
```

---

### Task 4C.2 — Move Test Harness to tests/ (G-013)

**Step 1**: Move `aneos_core/detection/artificial_neo_test_suite.py`
→ `tests/detection/test_artificial_neo_suite.py`

**Step 2**: Create `tests/detection/__init__.py` (empty) if it doesn't exist.

**Step 3**: Update any imports:
```python
# Old: from aneos_core.detection.artificial_neo_test_suite import ArtificialNEOTestSuite
# New: from tests.detection.test_artificial_neo_suite import ArtificialNEOTestSuite
```
Search all files: `grep -rn "artificial_neo_test_suite" .`

**Step 4**: Add `!tests/detection/test_artificial_neo_suite.py` exception to
`.gitignore` (the `test_*.py` pattern currently blocks it).

**Verification:**
```bash
grep -rn "artificial_neo_test_suite" aneos_core/
# Expected: 0 results
pytest tests/detection/ -q
```

---

### Task 4C.3 — Diagnose and Fix EMERGENCY Suppressions (G-009, G-022)

**File:** `aneos_core/analysis/advanced_scoring.py`

**Step 1**: Temporarily remove the two `# EMERGENCY` suppressions and run:
```bash
python -c "from aneos_core.analysis.advanced_scoring import AdvancedScoreCalculator; c = AdvancedScoreCalculator()"
```
Observe what warnings/errors appear.

**Step 2**: Based on what surfaces, likely causes and fixes:

_Likely cause A — config file not found_: `AdvancedScoringConfig` tries to
load `aneos_core/config/advanced_scoring_weights.json` but the path is wrong.
Fix: correct the path using `Path(__file__).parent.parent / "config" / "..."`.

_Likely cause B — weight normalization warning_: weights don't sum to 1.0.
Fix: normalize weights in `__post_init__` rather than suppressing.

_Likely cause C — logging initialization_: `logging.getLogger` called before
basicConfig. Fix: use module-level logger, not class-level.

**Step 3**: Once root cause is fixed, remove both `# EMERGENCY` comments.
Restore any configuration warnings that were suppressed — they are useful
for catching misconfiguration.

**Verification:**
```python
import warnings, logging
logging.basicConfig(level=logging.DEBUG)
from aneos_core.analysis.advanced_scoring import AdvancedScoreCalculator
c = AdvancedScoreCalculator()
# Expected: no warnings that require suppression; normal initialization logs
```

---

### Task 4C.4 — Designate Canonical Scorer (G-009)

**Step 1**: Run both scoring systems against a sample of the ground truth set
(10 naturals + 3 artificials) and compare F1 at the sigma-5 decision boundary.

```python
from aneos_core.analysis.scoring import ScoreCalculator
from aneos_core.analysis.advanced_scoring import AdvancedScoreCalculator
# Score each object with both; compare output classifications
```

**Step 2**: Record results. Designate the higher-F1 system as canonical.
Add an entry to `docs/architecture/ADR.md` updating ADR-008:

```markdown
### ADR-008 Update (2026-03-xx): Canonical Scoring System

**Decision**: [Standard/ATLAS] scoring system designated canonical.
**Empirical basis**: F1=[Standard_F1] (Standard) vs F1=[ATLAS_F1] (ATLAS)
on ground truth test set of 3 artificials + [N] naturals.
**Consequence**: [Non-canonical] system marked DEPRECATED in code.
```

**Step 3**: Mark the non-canonical class with a `# DEPRECATED` comment.

---

## Phase 4D — API and Model Quality

### Task 4D.1 — Create API Schema Layer (G-012)

**Create** `aneos_api/schemas/__init__.py` (exports all schemas).

**Create** `aneos_api/schemas/analysis.py`:
```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime

class IndicatorScores(BaseModel):
    orbital: float = 0.0
    velocity: float = 0.0
    temporal: float = 0.0
    geographic: float = 0.0
    physical: float = 0.0
    behavioral: float = 0.0

class AnalysisResponse(BaseModel):
    designation: str
    overall_score: float
    classification: str   # "NATURAL" | "ANOMALOUS" | "ARTIFICIAL"
    confidence: float
    indicator_scores: IndicatorScores
    risk_factors: List[str] = Field(default_factory=list)
    sigma_level: Optional[float] = None
    artificial_probability: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

**Create** `aneos_api/schemas/detection.py`:
```python
class DetectionResponse(BaseModel):
    designation: str
    is_artificial: bool
    artificial_probability: float
    sigma_level: float
    classification: str
    confidence: float
    evidence_count: int = 0
    interpretation: str = (
        "sigma_level measures statistical rarity under natural NEO null hypothesis. "
        "artificial_probability incorporates 0.1% base rate prior."
    )
```

**Create** `aneos_api/schemas/impact.py`:
```python
class ImpactResponse(BaseModel):
    designation: str
    collision_probability: float
    moon_collision_probability: Optional[float] = None
    moon_earth_ratio: Optional[float] = None
    impact_energy_mt: Optional[float] = None
    crater_diameter_km: Optional[float] = None
    risk_level: str   # "NEGLIGIBLE" | "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
    time_to_impact_years: Optional[float] = None
```

**Create** `aneos_api/schemas/health.py`:
```python
class CheckResult(BaseModel):
    status: str    # "ok" | "error" | "warning"
    detail: str

class HealthResponse(BaseModel):
    status: str
    checks: Dict[str, CheckResult]
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

**Update endpoints**: In `aneos_api/endpoints/analysis.py`,
`aneos_api/endpoints/enhanced_analysis.py`, `aneos_api/endpoints/monitoring.py`:
- Add `response_model=AnalysisResponse` (or relevant schema) to each route decorator
- Update return statements to construct the schema from the domain result

---

### Task 4D.2 — Generate and Commit OpenAPI Spec (G-014)

**Step 1**: Add to `Makefile`:
```makefile
spec:
	python -c "\
import sys; sys.path.insert(0, '.'); \
from aneos_api.app import create_app; \
import json; app = create_app(); \
print(json.dumps(app.openapi(), indent=2))" \
	> docs/api/openapi.json
```

**Step 2**: Run `make spec`; commit `docs/api/openapi.json`.

**Step 3**: Add CI drift detection in `.github/workflows/ci.yml`:
```yaml
- name: Verify OpenAPI spec up to date
  run: |
    make spec
    git diff --exit-code docs/api/openapi.json
```

**Step 4**: Update `docs/api/rest-api.md` header:
```markdown
> The authoritative API reference is `docs/api/openapi.json` (generated from
> FastAPI source). This file is kept for narrative context only.
```

---

### Task 4D.3 — Separate OrbitalElements from PhysicalProperties (G-011)

**This is the broadest change in Phase 4D. Do last. Run full test suite before
and after every step.**

**Step 1 — Identify all callsites reading physical fields from OrbitalElements**:
```bash
grep -rn "orbital_elements\.diameter\|orbital_elements\.albedo\|orbital_elements\.rot_per\|orbital_elements\.spectral_type\|oe\.diameter\|oe\.albedo\|oe\.spectral_type" aneos_core/ aneos_menu.py
```
Record every file and line.

**Step 2 — Verify PhysicalProperties already has these fields** (`models.py:430`).
If any field is missing from `PhysicalProperties`, add it.

**Step 3 — Update `SBDBSource._parse_sbdb_response()`** to populate
`physical_properties.diameter`, `physical_properties.albedo`, etc. instead of
`orbital_elements.diameter`.

**Step 4 — Update `DataFetcher._fetch_from_all_sources()`** to populate
`neo_data.physical_properties` (not `orbital_elements`) from fetch results.

**Step 5 — Update `indicators/physical.py`** to read from
`neo_data.orbital_elements.physical_properties` or directly via the NEOData
model — whichever is the correct path after Step 3.

**Step 6 — Remove fields from `OrbitalElements`**: Remove `diameter`, `albedo`,
`rot_per`, `spectral_type` fields and their validation logic.

**Step 7 — Run full test suite**. Fix every failure.

**Verification:**
```bash
grep -rn "orbital_elements\.diameter" aneos_core/ aneos_menu.py
# Expected: 0 results
pytest tests/ -q
# 54+ pass, 0 fail
```

---

## Phase 4E — Future Capabilities

### Task 4E.1 — Train ML Classifier (G-015)

Ground truth dataset (G-019 expanded) now provides the labelled training data.

**File:** `aneos_core/ml/features.py`

Implement `FeatureVector.from_ground_truth_object(obj)`:
```python
@classmethod
def from_ground_truth_object(cls, obj) -> "FeatureVector":
    oe = obj.orbital_elements
    pp = obj.physical_params or {}
    return cls(features=np.array([
        oe.get("a", 0.0),
        oe.get("e", 0.0),
        oe.get("i", 0.0),
        oe.get("omega", 0.0),
        oe.get("w", 0.0),
        pp.get("diameter", 0.0),
        pp.get("mass_estimate", 0.0),
        pp.get("albedo", 0.0),
    ], dtype=float))
```

**File:** `aneos_core/ml/training.py`

Implement `TrainingPipeline.train_from_ground_truth(objects)`:
- Extract feature vectors
- Fit `IsolationForest` (unsupervised anomaly) + `RandomForestClassifier` (supervised)
- Persist to `models/isolation_forest.joblib` and `models/random_forest.joblib`

**File:** `aneos_core/detection/detection_manager.py`

Add `DetectorType.ML = "ml"` and `MLDetectorWrapper` class. Priority 5 (lowest).

**Verification:**
```python
from aneos_core.detection.detection_manager import DetectionManager, DetectorType
d = DetectionManager(preferred_detector=DetectorType.ML)
result = d.analyze_neo({"a": 1.325, "e": 0.256, "i": 1.077})
print(result.artificial_probability)   # should not raise
```

---

### Task 4E.2 — Redis Health Check and Streaming Verify (G-016)

**File:** `aneos_core/utils/health.py`

Add `redis` key to `preflight_check()`:
```python
try:
    import redis as _redis
    r = _redis.from_url(os.environ.get("ANEOS_REDIS_URL", "redis://localhost:6379/0"),
                        socket_timeout=2)
    r.ping()
    results["redis"] = {"status": "ok", "detail": "PING successful"}
except Exception as exc:
    results["redis"] = {"status": "error", "detail": str(exc)}
```

**File:** `aneos_api/endpoints/streaming.py`

Add minimal Redis pub/sub on `/stream/health` endpoint:
```python
r.publish("aneos:events", json.dumps({"event": "health_ping",
                                       "timestamp": datetime.utcnow().isoformat()}))
```

---

### Task 4E.3 — Complete pickle → JSON Migration (G-017)

**File:** `aneos_core/data/cache.py`

Replace the pickle fallback path (lines ~282–321):
```python
# Instead of pickle.dumps(payload), use:
import dataclasses
if dataclasses.is_dataclass(payload):
    serialized = json.dumps(dataclasses.asdict(payload)).encode()
else:
    serialized = json.dumps(payload, default=str).encode()
cache_path.write_bytes(serialized)

# Instead of pickle.loads(data), use:
payload = json.loads(cache_path.read_bytes())
```

Add a one-time migration: at `CacheManager.__init__()`, scan the cache directory
and delete any files not ending in `.json`. Log a WARNING for each deleted file.

Remove `import pickle` from the file entirely.

---

### Task 4E.4 — Validate Chunk Boundary Overlap (G-018)

**Create** `tests/test_chunk_boundaries.py`:
```python
"""Test that chunk boundary objects are neither missed nor duplicated."""
from unittest.mock import patch
from aneos_core.polling.historical_chunked_poller import HistoricalChunkedPoller

def _make_object(designation, date):
    return {"designation": designation, "close_approach_date": date, "data": {}}

def test_boundary_objects_not_missed_or_duplicated():
    # Object at last day of chunk 1, first day of chunk 2, and in overlap window
    chunk1 = [_make_object("A", "2015-12-31"), _make_object("C", "2016-01-04")]
    chunk2 = [_make_object("B", "2016-01-01"), _make_object("C", "2016-01-04")]

    poller = HistoricalChunkedPoller.__new__(HistoricalChunkedPoller)
    merged = poller._merge_chunks([chunk1, chunk2])

    designations = [o["designation"] for o in merged]
    assert "A" in designations, "Boundary object A missed"
    assert "B" in designations, "Boundary object B missed"
    assert "C" in designations, "Overlap object C missed"
    assert designations.count("C") == 1, f"Duplicate C: {designations.count('C')}"
```

If `_merge_chunks` does not exist, add it as a deduplication method in
`HistoricalChunkedPoller` that removes duplicate designations keeping the
entry with more non-None fields.

Add `!tests/test_chunk_boundaries.py` to `.gitignore` exceptions.

---

### Task 4E.5 — Audit AI Annotator Language (G-025)

**File:** `aneos_core/reporting/ai_validation.py`

Read the `generate()` / `annotate()` methods. Search for:
- Any text containing "high confidence", "certain", "definitive", "confirmed"
- Any percentage claims derived from sigma level (not Bayesian posterior)

For each occurrence:
- If it states/implies high probability from sigma alone → rephrase to
  "statistically anomalous (σ=X)" and add "Bayesian P(artificial)=[Y]%"
- If it uses "confirmed artificial" for σ≥5 objects → change to
  "flagged as statistically anomalous"

Add a standard disclaimer constant to the module:
```python
_INTERPRETATION_DISCLAIMER = (
    "Statistical significance (σ) measures rarity under natural NEO hypothesis. "
    "Bayesian P(artificial) incorporates 0.1% base rate. "
    "Definitive classification requires propulsion or manoeuvre evidence."
)
```

Append this to every generated annotation.

---

### Task 4E.6 — Begin aneos_menu.py Decomposition (G-023)

This is the longest task. Do it iteratively — one module per PR, not all at once.

**Phase 4E.6a — Extract system/admin functions** (~80 functions):
Create `aneos_core/ui/system_menu.py`. Move:
- `display_system_status()`, `database_status()`, `view_mission_alerts()`
- `alert_management()`, `performance_metrics()`, `system_diagnostics()`
- `metrics_export()`, `configure_monitoring()`
- `database_management()`, `system_cleanup()`, `configuration_management()`
- `user_management()`, `system_maintenance()`, `system_reset()`
- `run_tests()`

In `aneos_menu.py`, replace method bodies with delegation:
```python
def display_system_status(self):
    from aneos_core.ui.system_menu import SystemMenu
    SystemMenu(self.console, self.config).display_system_status()
```

**Phase 4E.6b — Extract analysis display functions** (~60 functions):
Create `aneos_core/ui/analysis_display.py`. Move result-rendering functions.

**Phase 4E.6c — Extract detection result formatting** (~40 functions):
Create `aneos_core/ui/detection_display.py`.

After each extraction, run `pytest tests/ -q` to confirm nothing broke.
After Phase 4E.6a, add `tests/test_system_menu.py` with basic instantiation tests.

---

## Verification Checkpoints

**After Phase 4A (tasks 4A.1–4A.3):**
```python
from aneos_core.datasets.ground_truth_dataset_preparation import GroundTruthDatasetBuilder
from aneos_core.datasets.ground_truth_validator import GroundTruthValidator
from aneos_core.detection.detection_manager import DetectionManager, DetectorType

b = GroundTruthDatasetBuilder()
b.compile_verified_artificial_objects()
b.query_jpl_sbdb_natural_neos(limit=20)
detector = DetectionManager(preferred_detector=DetectorType.VALIDATED)
report = GroundTruthValidator().calibrated_run(b.artificial_objects + b.natural_objects, detector)
print(f"Corpus: {b.artificial_objects} artificials, {len(b.natural_objects)} naturals")
print(f"sens={report.sensitivity:.2f}, spec={report.specificity:.2f}, AUC={report.roc_auc:.3f}")
# Expected: ≥9 artificials, sens≥0.67, spec≥0.95
```

**After Phase 4B (tasks 4B.1–4B.4):**
```bash
grep -h "sources_used" neo_data/results/*.json 2>/dev/null | python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line.split(':', 1)[1].strip())
    print(d)" 2>/dev/null | head -5
# Expected: at least some results showing NEODyS or Horizons in sources_used
```

**After Phase 4C (tasks 4C.1–4C.4):**
```bash
ls aneos_core/detection/
# Expected: __init__.py, _archive/, detection_manager.py,
#           validated_sigma5_artificial_neo_detector.py
ls aneos_core/detection/ | grep -v _archive | wc -l
# Expected: 3 (plus the _archive/ directory = 4 entries total)
grep -rn "EMERGENCY" aneos_core/analysis/advanced_scoring.py
# Expected: 0 results
pytest tests/ -q
# Expected: 54+ pass, 0 fail
```

**After Phase 4D (tasks 4D.1–4D.3):**
```bash
ls aneos_api/schemas/
# Expected: __init__.py, analysis.py, detection.py, impact.py, health.py
grep -rn "orbital_elements\.diameter\|orbital_elements\.albedo" aneos_core/ aneos_menu.py
# Expected: 0 results
make spec && git diff --exit-code docs/api/openapi.json
# Expected: exit 0 (no drift)
pytest tests/ -q
# Expected: 54+ pass, 0 fail
```

**After Phase 4E (tasks 4E.1–4E.6):**
```bash
python -c "import pickle; print(open('aneos_core/data/cache.py').read())" | grep -c pickle
# Expected: 0 (or only in comments)
pytest tests/ -q
# Expected: all pass
wc -l aneos_menu.py
# Expected: materially fewer than 10,781 (target < 7,000 after 4E.6a-b)
```

---

## Files Modified / Created (complete list)

| Action | Path | Tasks |
|--------|------|-------|
| MODIFY | `aneos_core/datasets/ground_truth_dataset_preparation.py` | 4A.1 |
| MODIFY | `README.md` | 4A.2 |
| MODIFY | `docs/engineering/sigma5_success_criteria.md` | 4A.2 |
| MODIFY | `aneos_menu.py` | 4A.2, 4A.3, 4B.4, 4E.6 |
| MODIFY | `aneos_core/data/sources/neodys.py` | 4B.1 |
| MODIFY | `aneos_core/data/sources/mpc.py` | 4B.2 |
| MODIFY | `aneos_core/data/fetcher.py` | 4B.3 |
| CREATE | `aneos_core/detection/_archive/__init__.py` | 4C.1 |
| MOVE   | `detection/[4 files]` → `detection/_archive/` | 4C.1 |
| MODIFY | `aneos_core/detection/detection_manager.py` | 4C.1, 4E.1 |
| MOVE   | `detection/artificial_neo_test_suite.py` → `tests/detection/` | 4C.2 |
| MODIFY | `aneos_core/analysis/advanced_scoring.py` | 4C.3, 4C.4 |
| MODIFY | `docs/architecture/ADR.md` | 4C.4 |
| CREATE | `aneos_api/schemas/__init__.py` | 4D.1 |
| CREATE | `aneos_api/schemas/analysis.py` | 4D.1 |
| CREATE | `aneos_api/schemas/detection.py` | 4D.1 |
| CREATE | `aneos_api/schemas/impact.py` | 4D.1 |
| CREATE | `aneos_api/schemas/health.py` | 4D.1 |
| MODIFY | `aneos_api/endpoints/analysis.py` | 4D.1 |
| MODIFY | `Makefile` | 4D.2 |
| CREATE | `docs/api/openapi.json` | 4D.2 |
| MODIFY | `.github/workflows/ci.yml` | 4D.2 |
| MODIFY | `aneos_core/data/models.py` | 4D.3 |
| MODIFY | `aneos_core/analysis/indicators/physical.py` | 4D.3 |
| MODIFY | `aneos_core/data/sources/sbdb.py` | 4D.3 |
| MODIFY | `aneos_core/ml/features.py` | 4E.1 |
| MODIFY | `aneos_core/ml/training.py` | 4E.1 |
| MODIFY | `aneos_core/ml/prediction.py` | 4E.1 |
| MODIFY | `aneos_core/utils/health.py` | 4E.2 |
| MODIFY | `aneos_api/endpoints/streaming.py` | 4E.2 |
| MODIFY | `aneos_core/data/cache.py` | 4E.3 |
| CREATE | `tests/test_chunk_boundaries.py` | 4E.4 |
| MODIFY | `aneos_core/reporting/ai_validation.py` | 4E.5 |
| CREATE | `aneos_core/ui/system_menu.py` | 4E.6 |
| CREATE | `aneos_core/ui/analysis_display.py` | 4E.6 |
| CREATE | `aneos_core/ui/detection_display.py` | 4E.6 |
| CREATE | `tests/detection/test_artificial_neo_suite.py` | 4C.2 |
| CREATE | `tests/detection/__init__.py` | 4C.2 |
| MODIFY | `.gitignore` | 4C.2, 4E.4 |

---

## Sub-Phase Gate Definitions

| Gate | Pass Criteria |
|------|--------------|
| G-4A | ≥9 artificial objects in corpus; Bayesian ceiling documented in README + sigma5 doc; startup preflight prompts on failure |
| G-4B | ≥2 data sources return data in a live analysis run; `_generate_orbital_history` returns ≥3 epochs |
| G-4C | `detection/` has ≤4 entries; 0 EMERGENCY comments; canonical scorer documented in ADR-008 |
| G-4D | `aneos_api/schemas/` has 4 files; `openapi.json` committed; `OrbitalElements` has 0 physical fields |
| G-4E | 0 pickle imports in cache.py; chunk boundary test passes; ML detector registered; `aneos_menu.py` < 8000 lines |
